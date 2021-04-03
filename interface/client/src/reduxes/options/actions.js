import { createActions } from 'redux-actions'

const { changeOption } = createActions({
    CHANGE_OPTION: (issue, option) => (
        { issue, option }
    ),
})

export { changeOption }