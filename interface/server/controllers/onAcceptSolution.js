import Sequelize from 'sequelize'
import issues from '../issues'
import logger from '../logger'

const Op = Sequelize.Op

const calcScore = (user, solution) => {
    const utilities = user.utilities
    const options = solution.body

    const utilityScores = utilities.map(utility => {
        if (utility.type === 'INTEGER') {
            const selectedOptionValue = options[utility.name]
            // IntegerIssue
            if (user.role === 'recruiter') {
                return utility.weight * (utility.max - selectedOptionValue) / (utility.max - utility.min)
            } else {
                return utility.weight * (selectedOptionValue - utility.min) / (utility.max - utility.min)
            }
        } else if (utility.relatedTo) {
            const selectedOptionName = options[utility.name]
            const selectedOption = utility.options.find(o => o.name === selectedOptionName)
            // DependentDiscreteIssue
            const relatedIssue = utilities.find(u => u.name === utility.relatedTo)
            const relatedOptionName = options[relatedIssue.name]

            return utility.weight * selectedOption.biasedWeights[relatedOptionName]
        } else {
            const selectedOptionName = options[utility.name]
            // IndependentDiscreteIssue
            return utility.weight * utility.options.find(o => o.name === selectedOptionName).weight

        }
    })
    return utilityScores.reduce((p, c) => p + c)
}

const onAcceptSolution = (io, socket, db, payload) => {
    const { solutionId } = payload
    let currentUser, acceptedSolution

    const findCurrentUser = () =>
        db.user.find({ where: { socket_id: socket.id } })
            .then(cu => { currentUser = cu })

    const findAcceptedSolution = () =>
        db.solution.find({ where: { id: solutionId } })
            .then(s => { acceptedSolution = s })

    Promise.all([findCurrentUser(), findAcceptedSolution()]).then(() => {
        if (acceptedSolution.user_id === currentUser.id) {
            return
        }

        db.room.update(
            { status: 'completed' },
            { where: { id: currentUser.room_id } }
        )

        db.solution.update(
            { accepted: true },
            { where: { id: acceptedSolution.id } }
        )

        db.user.findAll({ where: { room_id: currentUser.room_id } })
            .then(roomUsers => {
                roomUsers
                    .filter(u => u.socket_id in io.sockets.connected)
                    .map(u => io.sockets.connected[u.socket_id]
                        .emit('completeNegotiation', {
                            solution: acceptedSolution.body,
                            score: calcScore(u, acceptedSolution),
                        }))
            })
    })
}

export default onAcceptSolution