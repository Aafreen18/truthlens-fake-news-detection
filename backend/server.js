import express from "express"
import cors from "cors"
import predictRouter from "./routes/predict.js"

const app = express()

app.use(cors({
    origin: "*"
}))

app.use(express.json())
app.use(predictRouter)

app.get("/", (req, res) => {
    res.send("Fake News Predictor API running")
})

const PORT = process.env.PORT || 3001

app.listen(PORT, () => {
    console.log(`Server listening on port ${PORT}`)
})