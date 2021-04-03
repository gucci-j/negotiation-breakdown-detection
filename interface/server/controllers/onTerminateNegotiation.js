import Sequelize from 'sequelize'
import issues from '../issues'
import logger from '../logger'

const Op = Sequelize.Op

const onTerminateNegotiation = (io, socket, db, payload) => {
    db.user.find({ where: { socket_id: socket.id } })
        .then(currentUser => {
            db.room.update(
                { status: 'terminated' },
                { where: { id: currentUser.room_id } }
            )
            return db.user.findAll({ where: { room_id: currentUser.room_id } })
        })
        .then(roomUsers => roomUsers
            .filter(u => u.socket_id in io.sockets.connected)
            .map(u => io.sockets.connected[u.socket_id].emit('terminateNegotiation', {}))
        )

}

export default onTerminateNegotiation