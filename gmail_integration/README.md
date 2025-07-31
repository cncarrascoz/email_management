# Email Management System

## Gmail Integration Overview

This component is, for now, a CLI implementation that integrates the ML email classification system with Gmail to bring intelligent and fast labeling into real inboxes. Using the Gmail API and OAuth2, the system can process fetched emails, run inference using the trained models, and apply suggested labels to emails.

> If you are looking for the ML classification system, please refere to the [project folder](../project/).

## Features

- OAuth2-based authentication via Google APIs
- Inbox reading and parsing
- Label detection and creation
- Serving predictions from trained ML models
- Future: Friendlier UI for reviewing predicted labels
