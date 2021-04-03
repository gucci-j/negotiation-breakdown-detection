import Express from 'express'
import compression from 'compression'
import path from 'path'

const app = Express()

app.use(compression({}))
app.use((req, res, next) => {
    res.header("Access-Control-Allow-Origin", "*")
    res.header("Access-Control-Allow-Headers", "Origin, X-Requested-With, Content-Type, Accept")
    next()
})
app.use(Express.static(path.resolve(__dirname, '../client/build')))

export default app