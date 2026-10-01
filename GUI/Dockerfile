ARG nginx_version=nginx:1.25.4-alpine

FROM node:22.0.0-alpine AS image
WORKDIR /usr/rag-search
COPY ./package*.json ./

FROM image AS build
RUN npm install --legacy-peer-deps
COPY . .
RUN ./node_modules/.bin/vite build
VOLUME /usr/rag-search

FROM $nginx_version AS web
COPY ./nginx/http-nginx.conf /etc/nginx/conf.d/default.conf
COPY --from=build ./usr/rag-search/build/assets /usr/share/nginx/html/rag-search/rag-search/assets
COPY --from=build ./usr/rag-search/build/index.html /usr/share/nginx/html/rag-search
COPY --from=build ./usr/rag-search/build/favicon.ico /usr/share/nginx/html/rag-search
RUN apk add --no-cache bash
COPY ./nginx/scripts/env.sh /docker-entrypoint.d/env.sh
RUN chmod +x /docker-entrypoint.d/env.sh
EXPOSE 3001

ENTRYPOINT [ "bash", "/docker-entrypoint.d/env.sh" ]
