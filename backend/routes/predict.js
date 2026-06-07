import express from "express"
import { runPrediction } from "../services/pythonServices.js"

const router = express.Router()

router.post("/predict", async (req, res) => {

    try {

        const { title, text } = req.body

        const safeTitle =
            typeof title === "string"
                ? title.trim()
                : ""

        const safeText =
            typeof text === "string"
                ? text.trim()
                : ""

        if (!safeTitle && !safeText) {
            return res.status(400).json({
                error: "Provide title or article text"
            })
        }

        const combined =
            `${safeTitle} ${safeText}`

        const result =
            await runPrediction(combined)

        return res.json(result)

    } catch (err) {

        console.error(
            "Prediction Error:",
            err
        )

        return res.status(500).json({
            error: "Server Error"
        })

    }

})

export default router