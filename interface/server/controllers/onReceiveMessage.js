import Sequelize from 'sequelize'
import logger from '../logger'
import moment from 'moment'
import taskConfig from '../config/task'

const Op = Sequelize.Op

const onReceiveMessage = (io, socket, db, payload) => {

    db.user
        .find({ where: { socket_id: socket.id } })
        .then(currentUser => db.comment
            .create({
                body: payload.body,
                user_id: currentUser.id,
            }))
        .then(message => {
            db.user
                .find({ where: { socket_id: socket.id } })
                .then(currentUser => db.user
                    .findAll({
                        where: { room_id: currentUser.room_id },
                    })
                )
                .then(roomUsers => {
                    roomUsers
                        .filter(u => u.socket_id in io.sockets.connected)
                        .map(u => {
                            io.sockets.connected[u.socket_id]
                                .emit('receiveMessage', {
                                    body: payload.body,
                                    fromYourself: socket.id === u.socket_id,
                                    id: message.id,
                                })
                        })
                })
        })
}

export default onReceiveMessage