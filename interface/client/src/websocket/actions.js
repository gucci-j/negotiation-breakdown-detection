import { createActions } from 'redux-actions'

const websocketActionCreator = (messageType, messageData, reducer = false) => (
    (dispatch, getState, { emit }) => {
        emit(messageType, messageData)
        if (reducer) {
            return dispatch({
                type: messageType,
                payload: messageData,
            })
        }
    }
)

const emitJoinLobby = (assignmentId, workerId) => (
    websocketActionCreator('joinLobby', { assignmentId, workerId }, true)
)

const emitSendMessage = payload => (
    websocketActionCreator('sendMessage', payload, true)
)

const emitSendSolution = solution => (
    websocketActionCreator('sendSolution', { solution }, true)
)

const emitAcceptSolution = solutionId => (
    websocketActionCreator('acceptSolution', { solutionId: solutionId }, true)
)

const emitTerminateNegotiation = () => (
    websocketActionCreator('terminateNegotiation', {}, true)
)

const {
    joinRoom,
    joinLobby,
    receiveMessage,
    leaveRoom,
    sendSolution,
    showResult,
    receiveSolution,
    completeNegotiation,
    terminateNegotiation,
} = createActions({
    joinRoom: payload => payload,
    joinLobby: payload => { payload },
    receiveMessage: payload => payload,
    leaveRoom: () => { },
    sendSolution: payload => payload,
    showResult: payload => payload,
    receiveSolution: payload => payload,
    completeNegotiation: payload => payload,
    terminateNegotiation: payload => payload,
})

export {
    emitJoinLobby,
    emitSendMessage,
    emitSendSolution,
    emitAcceptSolution,
    emitTerminateNegotiation,
    joinRoom,
    joinLobby,
    sendSolution,
    receiveMessage,
    leaveRoom,
    showResult,
    receiveSolution,
    completeNegotiation,
    terminateNegotiation,
}