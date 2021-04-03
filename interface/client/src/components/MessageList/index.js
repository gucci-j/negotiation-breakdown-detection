import MessageList from './MessageList'
import { connect } from 'react-redux'
import { emitAcceptSolution } from '../../websocket/actions'

const mapStateToProps = (state) => ({
    utilities: state.utilities.toJS(),
    options: state.options.toJS(),
    messages: state.messages,
    role: state.role,
})

const mapDispatchToProps = (dispatch) => ({
    acceptSolution: solutionId => dispatch(emitAcceptSolution(solutionId))
})

export default connect(mapStateToProps, mapDispatchToProps)(MessageList)