FROM node:20-alpine as build

WORKDIR /app

COPY frontend/package.json frontend/vite.config.js /app/
COPY frontend/index.html /app/index.html
COPY frontend/src /app/src

RUN npm install && npm run build

FROM nginx:alpine

COPY --from=build /app/dist /usr/share/nginx/html

EXPOSE 80

CMD ["nginx", "-g", "daemon off;"]
