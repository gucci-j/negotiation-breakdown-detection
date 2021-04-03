import Talking from './Talking'
import { changeOption } from '../../reduxes/options/actions'
import { connect } from 'react-redux'
import { actions as wsActions } from '../../websocket'

const mapStateToProps = (store) => ({
    utilities: store.utilities.toJS(),
    options: store.options.toJS(),
    role: store.role,
    messages: store.messages,
})

const mapDispatchToProps = dispatch => ({
    sendMessage: (body) => (
        dispatch(wsActions.emitSendMessage({ body }))
    ),
    changeOption: (issue, option) => (
        dispatch(changeOption(issue, option))
    ),
    sendSolution: (options) => (
        dispatch(wsActions.emitSendSolution(options))
    ),
    terminateNegotiation: () => (
        dispatch(wsActions.emitTerminateNegotiation())
    )
})

export default connect(mapStateToProps, mapDispatchToProps)(Talking)