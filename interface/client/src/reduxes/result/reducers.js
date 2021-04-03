import {
    completeNegotiation,
    terminateNegotiation,
} from "../../websocket/actions"

import { default as Im } from "immutable"
import { handleActions } from "redux-actions"
import calcReward from "../../utils/calcReward"

const ResultRecord = Im.Record({
    succeeded: null,
    score: null,
    reward: null,
})

const defaultState = new ResultRecord({
    succeeded: null,
    score: null,
    reward: null,
})

const reducer = handleActions(
    {
        [completeNegotiation]: (state, { payload: { score } }) =>
            state
                .set("succeeded", true)
                .set("score", parseInt(score * 1000) / 10)
                .set("reward", "$" + calcReward(score * 100)),
        [terminateNegotiation]: (state, { payload }) =>
            state.set("succeeded", false).set("reward", "$0.0"),
    },
    defaultState,
)

export default reducer
