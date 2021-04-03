const calcScore = (options, utilities, role) => {
    // console.log(Im.fromJS(options).toJS())
    // console.log(Im.fromJS(utilities).toJS())
    const utilityScores = utilities.map(utility => {
        if (utility.type === "INTEGER") {
            const selectedOptionValue = options[utility.name]
            // IntegerIssue
            if (role === "recruiter") {
                return (
                    (utility.weight * (utility.max - selectedOptionValue)) /
                    (utility.max - utility.min)
                )
            } else {
                return (
                    (utility.weight * (selectedOptionValue - utility.min)) /
                    (utility.max - utility.min)
                )
            }
        } else if (utility.relatedTo) {
            const selectedOptionName = options[utility.name]
            const selectedOption = utility.options.find(
                o => o.name === selectedOptionName,
            )
            // DependentDiscreteIssue
            const relatedIssue = utilities.find(
                u => u.name === utility.relatedTo,
            )
            const relatedOptionName = options[relatedIssue.name]

            return (
                utility.weight * selectedOption.biasedWeights[relatedOptionName]
            )
        } else {
            const selectedOptionName = options[utility.name]
            // IndependentDiscreteIssue
            return (
                utility.weight *
                utility.options.find(o => o.name === selectedOptionName).weight
            )
        }
    })
    return utilityScores.reduce((p, c) => p + c)
}

export default calcScore
