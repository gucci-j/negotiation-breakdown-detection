import Sequelize from 'sequelize'
import issues from '../issues'
import logger from '../logger'

const Op = Sequelize.Op

const onReceiveSolution = (io, socket, db, payload) => {
    db.user.update(
        { solution: JSON.stringify(payload.solution) },
        { where: { socket_id: socket.id } },
    )
        .then(() => db.user.find({ where: { socket_id: socket.id } }))
        .then(currentUser => {
            db.solution.create({
                body: JSON.stringify(payload.solution),
                user_id: currentUser.id,
            })
                .then(solution => {
                    db.user.findAll({ where: { room_id: currentUser.room_id } })
                        .then(roomUsers => {
                            roomUsers
                                .filter(u => u.socket_id in io.sockets.connected)
                                .map(u => {
                                    io.sockets.connected[u.socket_id]
                                        .emit('receiveSolution', {
                                            createdAt: solution.created_at,
                                            id: solution.id,
                                            solution: solution.body,
                                            fromYourself: socket.id === u.socket_id,
                                        })
                                })
                        })
                })
        })
}

export default onReceiveSolution