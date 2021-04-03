import { handleActions } from 'redux-actions'
import { joinRoom } from '../../websocket/actions'

const defaultState = null

const reducer = handleActions(
    {
        [joinRoom]: (state, { payload }) =>
            payload.role
        ,
    },
    defaultState
)

export { reducer as default }
