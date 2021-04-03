Job Interview Negotiation Platform
===

## database

### init

```
brew install postgres
initdb ./database --username=test
postgres -D ./database
createdb -U test mturk_dev
psql --user=test --dbname=mturk_dev
```

### start

```
pg_ctl -D ./database -l logfile start
```

## server

### init

```
cd ./server
npm install
npm run db:reset
npm run db:migrate
```

### start

```
cd ./server
npm run start
```

## client

### init

```
cd ./client
npm install
```

## start

```
cd ./client
npm run start
```

## Note
When you run this interface, please give a worker ID for each negotiator.

Example:
```
http://localhost:3000/?workerId=0
```