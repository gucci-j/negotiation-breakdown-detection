import Sequelize from 'sequelize'
import moment from 'moment'
import taskConfig from '../config/task'

const Op = Sequelize.Op

const onDisconnect = (io, socket, db) => {
    db.user
        .find({ where: { socket_id: socket.id } })
        .then(currentUser => {
            if (currentUser.room_id) {
                db.user
                    .findAll({ where: { room_id: currentUser.room_id } })
                    .then(restUsers => {

                        if (restUsers.filter(u => u.solution).length < restUsers.length) {

                            restUsers.map(u => u.destroy())
                            restUsers
                                .filter(u => u.socket_id in io.sockets.connected)
                                .map(u => {
                                    io.sockets.connected[u.socket_id]
                                        .emit('leaveRoom', { createdAt: Date.now(), })
                                })

                            db.room.update(
                                { status: 'aborted' },
                                { where: { id: currentUser.room_id, status: 'talking' } }
                            )
                        }
                    })
                    .then(() => {
                        currentUser.destroy()
                    })
            } else {
                currentUser.destroy()
            }
        })
}

export default onDisconnect