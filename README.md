# KnowCo – Interactive AI for Employee Onboarding

## Problem
Learning is often time-consuming, and people have different preferences. Some enjoy reading, others prefer listening, and many learn best by asking questions and challenging themselves.  
No matter the situation—school, university, or workplace—continuous learning is part of life.  

In this application, we focus on **employee onboarding**. However, the approach can easily be extended to many other contexts.  

Onboarding new employees is typically **slow, costly, and disengaging**. Every year, departments face high turnover and long ramp-up times, with new hires spending weeks reading documents or shadowing colleagues before they feel confident in their roles.  

## Solution
Our solution is an application that enables new employees to learn about the company interactively and at their own pace, making the onboarding process faster, more engaging, and more effective.  

**KnowCo** is an AI-powered onboarding assistant that makes the process interactive, personalized, and fast.  

Instead of spending hours with static training manuals, new hires can chat or speak directly with a voice AI that is connected to the company’s internal documents. The assistant:  
- **Generates questions** about the company, and employees can answer using either voice or text.  
- **Answers employee questions** in text or voice—depending on their preference.  
- **Provides help** if employees don’t know the answer, offering clear explanations.  
- **Supports direct Q&A with documents.**  

Our current prototype focuses on **general company culture and department-level knowledge**. However, the system is designed to expand into **technical training, HR processes, and even simulated customer interactions** (e.g., practicing how to respond to an angry customer call) in the future.  

## How It Works (Tech Overview)
- **Document Processing** – Convert PDFs to text, extract tables, and generate image descriptions with AI (prompt engineering).  
- **Knowledge Base** – Combine all content (text, tables, image descriptions), apply clustering-based chunking, and create embeddings stored in a vector database.  
- **Retrieval & Q&A** – Use prompt engineering + RAG to find the best answers from company knowledge.  
- **Training & Coaching** – AI generates quiz questions from documents, with answers available in text or voice.  
- **Voice Interface** – Real-time dialogue via Speech-to-Text and Text-to-Speech (ElevenLabs).  

## Impact
With **KnowCo**, onboarding time can shrink from **weeks to days**.  
Trainees gain confidence faster, managers save time, and companies foster stronger connections with new employees.  

## Roadmap
- Expand simulations to technical support and product teams.  
- Add personalized learning paths and progress tracking.  
- Integrate with HR systems and CRMs for seamless rollout.  

## Team
- **Najmeh Akbari**  
- **Farnaz Delirie**  
