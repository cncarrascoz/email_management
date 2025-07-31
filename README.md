# Email Management System

## Project Overview

This repository contains an intelligent email management system that combines:
- A **machine learning model** that classifies emails into relevant categories
- A **Gmail integration pipeline** that connects the model to real inboxes

The project is built to help users organize their inboxes automatically, reducing manual email classification and improving productivity.

### Components

- [**ML Classification System** (`project/`)](project/)
  - Trained on the Enron dataset
  - Uses TF-IDF and DistilBERT
  - Includes active learning and manual labeling
- [**Gmail Integration** (`gmail_integration/`)](gmail_integration/)
  - OAuth2 authentication
  - Inbox reading and labeling
  - Intended to serve ML predictions on real emails

> To get started quickly with either part, please refer to their respective folders.

## Getting Started

Each component has its own README and setup instructions:
- [ML System Setup](project/)
- [Gmail Integration Setup](gmail_integration/)

## Vision

The long-term goal is to build a unified system where the ML model can organize a user's inbox according to their preferences.
