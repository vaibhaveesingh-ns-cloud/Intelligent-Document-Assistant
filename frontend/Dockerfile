FROM node:20-alpine as build

WORKDIR /app

# Accept Vite API base path at build time
ARG VITE_API_BASE=/api
ENV VITE_API_BASE=${VITE_API_BASE}

# With compose build context set to ./frontend, copy files without prefix
COPY package.json vite.config.js /app/
COPY index.html /app/index.html
COPY src /app/src

RUN npm install && npm run build

FROM nginx:alpine

COPY --from=build /app/dist /usr/share/nginx/html
COPY nginx.conf /etc/nginx/conf.d/default.conf

EXPOSE 80

CMD ["nginx", "-g", "daemon off;"]
