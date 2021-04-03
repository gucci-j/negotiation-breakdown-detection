import { default as Im } from 'immutable'
import { handleActions } from 'redux-actions'
import { joinRoom } from '../../websocket/actions'

const defaultState = new Im.List()

const reducer = handleActions(
    {
        [joinRoom]: (state, { payload }) => (
            Im.fromJS(payload.utilities)
        )
        ,
    },
    defaultState
)

export { reducer as default }
