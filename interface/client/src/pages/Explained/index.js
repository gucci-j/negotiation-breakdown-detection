import Explained from './Explained'
import { connect } from 'react-redux'
import pageNameTypes from '../../reduxes/pages/pageNameTypes'
import { setVisiblePage } from '../../reduxes/pages/actions'
import { actions as wsActions } from '../../websocket'

const mapStateToProps = () => ({})

const mapDispatchToProps = dispatch => ({
  forwardPage: (assignmnentId, workerId) => (
    dispatch(wsActions.emitJoinLobby(assignmnentId, workerId))
  )
})

export default connect(mapStateToProps, mapDispatchToProps)(Explained)