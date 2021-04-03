const score2emoji = score =>
    score < 50
        ? "😰 Very Poor"
        : score < 60
        ? "😥 Poor"
        : score < 70
        ? "😞 Not so good"
        : score < 80
        ? "😃 Good"
        : score < 90
        ? "😄 Very Good"
        : "😁 Supreme"

export default score2emoji
