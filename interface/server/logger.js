import log4js from 'log4js'

const logger = {
    system: log4js.getLogger('system'),
    access: log4js.getLogger('access'),
    error: log4js.getLogger('error')
}

Object.keys(logger).map(k => {
    logger[k].level = 'ALL'
})

export default logger