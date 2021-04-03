import Ending from './Ending'
import { connect } from 'react-redux'
import pageNameTypes from '../../reduxes/pages/pageNameTypes'
import { setVisiblePage } from '../../reduxes/pages/actions'
import { actions as wsActions } from '../../websocket'

const mapStateToProps = store => ({
    result: store.result.toJS(),
    utilities: store.utilities.toJS(),
    options: store.options.toJS(),
    role: store.role,
})

const mapDispatchToProps = dispatch => ({
})

export default connect(mapStateToProps, mapDispatchToProps)(Ending)