import React from 'react'

const ScoreBadge = ({ weight, style, withText = false }) => {
    const bgClassName = weight > 0.3 ? 'ScoreBadge-veryhigh' : weight > 0.2 ? 'ScoreBadge-high' : weight > 0.1 ? 'ScoreBadge-low' : 'ScoreBadge-verylow'
    const text = weight > 0.3 ? 'Very High' : weight > 0.2 ? 'High' : weight > 0.1 ? 'Low' : 'Very Low'

    return (
        <span
            className={`ScoreBadge ${bgClassName}`}
            style={{
                ...style,
            }}
        >
            {parseInt(weight * 100)}
            {
                withText &&
                ` (${text})`
            }
        </span>
    )
}

export default ScoreBadge