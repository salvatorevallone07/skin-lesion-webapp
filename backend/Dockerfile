# syntax=docker/dockerfile:1
# Build context = project ROOT (see docker-compose.yml) so we can copy both
# frontend/ and backend/.

# ---------- Stage 1: build the Angular 21 SPA ----------
FROM node:24 AS frontend
WORKDIR /ui
COPY frontend/package.json frontend/package-lock.json* ./
RUN npm install
COPY frontend/ ./
# Outputs static files to /ui/dist/skin-lesion-frontend/browser
RUN npm run build -- --configuration production

# ---------- Stage 2: build the Spring Boot WAR (Java 25) ----------
FROM maven:3.9-eclipse-temurin-25 AS backend
WORKDIR /app
COPY backend/pom.xml ./
RUN mvn -B -q dependency:go-offline
COPY backend/src ./src
# Drop the compiled SPA into the WAR's static resources
COPY --from=frontend /ui/dist/skin-lesion-frontend/browser/ ./src/main/resources/static/
RUN mvn -B -q clean package -DskipTests

# ---------- Stage 3: runtime on Tomcat 11 / JDK 25 ----------
FROM tomcat:11.0-jdk25-temurin
RUN rm -rf /usr/local/tomcat/webapps/*
COPY --from=backend /app/target/ROOT.war /usr/local/tomcat/webapps/ROOT.war
EXPOSE 8080
CMD ["catalina.sh", "run"]
