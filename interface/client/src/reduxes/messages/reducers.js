import { receiveMessage, receiveSolution } from '../../websocket/actions'

import { default as Im } from 'immutable'
import { handleActions } from 'redux-actions'

const MessageRecord = Im.Record({
    id: null,
    body: null,
    fromYourself: null,
})

const SolutionRecord = Im.Record({
    id: null,
    solution: null,
    fromYourself: null,
})

const defaultState = new Im.List()

const reducer = handleActions(
    {
        [receiveMessage]: (state, { payload }) => (
            state.push(new MessageRecord()
                .set('id', payload.id)
                .set('body', payload.body)
                .set('fromYourself', payload.fromYourself))
        ),
        [receiveSolution]: (state, { payload }) => (
            state.push(new SolutionRecord()
                .set('id', payload.id)
                .set('solution', payload.solution)
                .set('fromYourself', payload.fromYourself))
        ),
    },
    defaultState
)

export { reducer as default }
