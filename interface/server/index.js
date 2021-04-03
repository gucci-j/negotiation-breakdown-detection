import db, { Comment, Room, User } from './models'

import Http from 'http'
import Sequelize from 'sequelize'
import SocketIO from 'socket.io'
import app from './app'
import issues from './issues'
import logger from './logger'
import messageTypes from '../common/messageTypes'
import moment from 'moment'
import onAcceptSolution from './controllers/onAcceptSolution'
import onDisconnect from './controllers/onDisconnect'
import onJoinLobby from './controllers/onJoinLobby'
import onReceiveMessage from './controllers/onReceiveMessage'
import onReceiveSolution from './controllers/onReceiveSolution'
import onTerminateNegotiation from './controllers/onTerminateNegotiation'

const srv = Http.Server(app)
const io = new SocketIO(srv)

const PORT = process.env.PORT || 8080
const N_MAX_SCORE = 100

const Op = Sequelize.Op

db.user.destroy({ where: { deleted_at: { [Op.eq]: null } } })

io.set('heartbeat interval', 1000)
io.set('heartbeat timeout', 5000)
io.on('connection', socket => {
    const onConnect = (socket) => {
        User
            .create({
                socket_id: socket.id,
            })
    }
    onConnect(socket)

    /*
      joinLobby
  
      Let waiting users join in a room if the number of them is not less than config.task.N_USERS_IN_ROOM.
    */
    socket.on(messageTypes.joinLobby, payload => { onJoinLobby(io, socket, db, payload) })
    socket.on(messageTypes.sendMessage, payload => { onReceiveMessage(io, socket, db, payload) })
    socket.on(messageTypes.sendSolution, payload => { onReceiveSolution(io, socket, db, payload) })
    socket.on(messageTypes.acceptSolution, payload => { onAcceptSolution(io, socket, db, payload) })
    socket.on(messageTypes.terminateNegotiation, payload => { onTerminateNegotiation(io, socket, db, payload) })

    socket.on('disconnect', () => { onDisconnect(io, socket, db) })
})

srv.listen(PORT, () => {
    logger.system.info('Listening on *:' + PORT)
})