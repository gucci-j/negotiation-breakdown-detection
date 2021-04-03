import { createActions } from 'redux-actions'

const { setVisiblePage } = createActions({
    SET_VISIBLE_PAGE: (pageName) => ({ pageName }),
})

export { setVisiblePage }