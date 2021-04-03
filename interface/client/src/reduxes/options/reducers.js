import { default as Im } from 'immutable'
import { changeOption } from './actions'
import { handleActions } from 'redux-actions'
import { joinRoom } from '../../websocket/actions'

const defaultState = new Im.Map()

const reducer = handleActions(
    {
        [changeOption]: (state, { payload }) =>
            state.set(payload.issue, payload.option),
        [joinRoom]: (state, { payload }) => {
            return [state].concat(payload.utilities).reduce((prevState, issue) => {
                prevState = prevState || state
                if (issue.type === 'INTEGER') {
                    return prevState.set(issue.name, (issue.min + issue.max) / 2)
                } else {
                    return prevState.set(issue.name, issue.options[0].name)
                }
            })
        },
    },
    defaultState
)

export { reducer as default }
