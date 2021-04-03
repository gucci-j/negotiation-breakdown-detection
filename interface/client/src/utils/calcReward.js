const calcReward = score => (score < 50 ? 0.2 : 0.2 + ((score - 50) / 50) * 1.0)

export default calcReward
