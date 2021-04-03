export default (sequelize, DataTypes) => {
  const room = sequelize.define('room', {
    status: DataTypes.STRING
  }, {
      underscored: true,
      paranoid: true,
    })
  room.associate = function (models) {
    // associations can be defined here
  }
  return room
}