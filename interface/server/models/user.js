export default (sequelize, DataTypes) => {
  const user = sequelize.define('user', {
    assignment_id: DataTypes.STRING,
    worker_id: DataTypes.STRING,
    socket_id: DataTypes.STRING,
    room_id: DataTypes.INTEGER,
    is_bonus_paid: DataTypes.BOOLEAN,
    assignment_status: DataTypes.STRING,
    role: DataTypes.INTEGER,
    joined_at: DataTypes.STRING,
    utilities: DataTypes.STRING,
  }, {
      underscored: true,
      paranoid: true,
    })
  user.associate = function (models) {
    user.belongsTo(models.room)
  }
  return user
}