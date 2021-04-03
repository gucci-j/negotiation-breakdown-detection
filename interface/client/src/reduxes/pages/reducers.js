import { completeNegotiation, terminateNegotiation } from '../../websocket/actions'
import { joinLobby, joinRoom, leaveRoom, receiveMessage } from '../../websocket/actions'

import { default as Im } from 'immutable'
import { handleActions } from 'redux-actions'
import pageNameTypes from './pageNameTypes'
import { setVisiblePage } from './actions'

const PagesRecord = Im.Record({
    pageName: null
})

const defaultState = new PagesRecord({
    pageName: pageNameTypes.EXPLAINED
})

const reducer = handleActions(
    {
        [setVisiblePage]: (state, { payload: { pageName } }) =>
            state.set('pageName', pageName),
        [joinLobby]: (state, { payload }) =>
            state.set('pageName', pageNameTypes.WAITING),
        [joinRoom]: (state, { payload }) =>
            state.set('pageName', pageNameTypes.TALKING),
        [leaveRoom]: (state, { payload }) =>
            state.pageName === pageNameTypes.ENDING ?
                state :
                state.set('pageName', pageNameTypes.FAILED),
        [terminateNegotiation]: (state, { payload }) =>
            state.set('pageName', pageNameTypes.ENDING),
        [completeNegotiation]: (state, { payload }) =>
            state.set('pageName', pageNameTypes.ENDING),
        [receiveMessage]: (state, { payload }) => {
            return state
        },
    },
    defaultState
)

export default reducer