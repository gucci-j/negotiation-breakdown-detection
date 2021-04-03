import Failed from './Failed'
import { connect } from 'react-redux'
import pageNameTypes from '../../reduxes/pages/pageNameTypes'
import { setVisiblePage } from '../../reduxes/pages/actions'
import { actions as wsActions } from '../../websocket'

const mapStateToProps = () => ({})

const mapDispatchToProps = dispatch => ({
  forwardPage: () => (
    dispatch(wsActions.emitJoinLobby())
  )
})

export default connect(mapStateToProps, mapDispatchToProps)(Failed)