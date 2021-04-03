import Im from 'immutable'
import Sequelize from 'sequelize'
import issues from '../issues'
import logger from '../logger'
import moment from 'moment'
import taskConfig from '../config/task'

const Op = Sequelize.Op

const roles = ['recruiter', 'worker']

const shuffle = (a) => {
    var j, x, i
    for (i = a.length - 1; i > 0; i--) {
        j = Math.floor(Math.random() * (i + 1))
        x = a[i]
        a[i] = a[j]
        a[j] = x
    }
    return a
}

const onJoinLobby = (io, socket, db, payload) => {
    db.user
        .find({ where: { socket_id: socket.id } })
        .then(currentUser =>
            db.user.update(
                {
                    joined_at: moment.utc().format('YYYY-MM-DD HH:mm:ss'),
                    assignment_id: payload.assignmentId,
                    worker_id: payload.workerId,
                },
                { where: { id: currentUser.id } }
            )
        )
        .then(currentUser =>
            db.user.findAll({
                where: { room_id: { [Op.eq]: null }, joined_at: { [Op.ne]: null }, },
                order: [['id', 'DESC']],
            })
        )
        .then(waitingUsers =>
            waitingUsers
                .filter(u => u.socket_id in io.sockets.connected)
                .slice(0, taskConfig.N_USERS_IN_ROOM)
        )
        .then(joinableUsers => {
            if (joinableUsers.length >= taskConfig.N_USERS_IN_ROOM) {
                shuffle(roles)

                const joinableUserKeys = [...joinableUsers.keys()]
                joinableUserKeys.map(u => {
                    const role = roles[u]
                    let userIssues = Im.fromJS(issues).toJS()
                    let weights = Array.from({ length: issues.length }, () => Math.random())
                    let sumWeights = weights.reduce((acc, v) => (acc + v))
                    weights = weights.map(w => w / sumWeights)

                    joinableUsers[u].role = role
                    joinableUsers[u].utilities = [...userIssues.keys()]
                        .map(i => {
                            let issue = userIssues[i]
                            issue.weight = 0.1 + weights[i] * 0.5
                            if (issue.type === 'DISCRETE') {
                                let issueWeights = Array.from({ length: issue.options.length }, () => Math.random())
                                let maxIssueWeights = issueWeights.reduce((acc, v) => (acc > v ? acc : v))
                                let minIssueWeights = issueWeights.reduce((acc, v) => (acc < v ? acc : v))
                                issueWeights = issueWeights.map(w => (w - minIssueWeights) / (maxIssueWeights - minIssueWeights))
                                let optionKeys = [...issue.options.keys()]
                                optionKeys.map(o => {
                                    issue.options[o].weight = issueWeights[o]
                                    if (issue.relatedTo) {
                                        issue.options[o].bias = Object()
                                    }
                                })
                            }
                            return issue
                        })

                    joinableUsers[u].utilities.map(issue => {
                        if (issue.relatedTo) {
                            const relatedIssue = joinableUsers[u].utilities.find(i => i.name === issue.relatedTo)
                            const optionKeys = [...issue.options.keys()]
                            let issueWeights = Array.from({ length: issue.options.length }, () => Math.random())

                            optionKeys.map(o => {
                                issue.options[o].biasedWeights = new Object()
                            })

                            relatedIssue.options.map(ro => {
                                let biases = Array.from({ length: issue.options.length }, () => Math.random() * 0.5)
                                let biasedWeights = optionKeys.map(o => issueWeights[o] + biases[o])
                                // let biasedWeights = optionKeys.map(o => biases[o])
                                let maxIssueWeights = biasedWeights.reduce((acc, v) => (acc > v ? acc : v))
                                let minIssueWeights = biasedWeights.reduce((acc, v) => (acc < v ? acc : v))
                                biasedWeights = biasedWeights.map(w => (w - minIssueWeights) / (maxIssueWeights - minIssueWeights))

                                optionKeys.map(o => {
                                    issue.options[o].biasedWeights[ro.name] = biasedWeights[o]
                                })
                            })
                        }
                    })
                })

                db.sequelize
                    .transaction()
                    .then(t => (
                        db.room
                            .create({ status: 'talking' }, { transaction: t })
                            .then(room => (
                                db.user.update(
                                    { room_id: room.id },
                                    { where: { id: joinableUsers.map(u => u.id) }, transaction: t },
                                )
                            ))
                            .then(() => (
                                Promise.all(
                                    joinableUsers.map(u => (
                                        db.user.update(
                                            { role: u.role, utilities: JSON.stringify(u.utilities), },
                                            { where: { id: u.id }, transaction: t },
                                        )
                                    ))
                                )
                            ))
                            .then(function () {
                                t.commit()
                                joinableUsers
                                    .filter(u => u.socket_id in io.sockets.connected)
                                    .map(u => {
                                        io.sockets.connected[u.socket_id]
                                            .emit('joinRoom', {
                                                createdAt: Date.now(),
                                                utilities: u.utilities,
                                                role: u.role,
                                            })
                                    })
                            })
                            .catch(err => {
                                logger.system.error(err)
                                t.rollback()
                            })
                    ))
            }
        })
}

export default onJoinLobby