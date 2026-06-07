import { PythonShell } from "python-shell"
import path from "path"
import { fileURLToPath } from "url"

const __filename = fileURLToPath(import.meta.url)
const __dirname = path.dirname(__filename)

const PYTHON_SCRIPT_PATH = path.resolve(
    __dirname,
    "../../ml-model"
)

console.log(
    "Python Path:",
    PYTHON_SCRIPT_PATH
)

export const runPrediction = async (text) => {

    const options = {
        mode: "text",
        pythonPath: "python3",
        scriptPath: PYTHON_SCRIPT_PATH,
        args: [text]
    }

    try {

        const result = await PythonShell.run(
            "predict.py",
            options
        )

        if (!result || result.length === 0) {
            throw new Error(
                "No output from Python"
            )
        }

        const output = result[0].trim()

        if (output === "error") {
            throw new Error(
                "Prediction failed"
            )
        }

        return {
            prediction: output
        }

    } catch (err) {

        console.error(
            "Python Error:",
            err
        )

        throw err
    }
}