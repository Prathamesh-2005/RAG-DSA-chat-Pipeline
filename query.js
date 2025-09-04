import * as dotenv from 'dotenv';
dotenv.config();
import readlineSync from 'readline-sync';
import { GoogleGenerativeAIEmbeddings } from '@langchain/google-genai';
import { Pinecone } from '@pinecone-database/pinecone';
import { GoogleGenAI } from "@google/genai";

const ai = new GoogleGenAI({});
const History = []

async function transformQuery(question){

History.push({
    role:'user',
    parts:[{text:question}]
    })  

const response = await ai.models.generateContent({
    model: "gemini-2.0-flash",
    contents: History,
    config: {
      systemInstruction: `You are a query rewriting expert. Based on the provided chat history, rephrase the "Follow Up user Question" into a complete, standalone question that can be understood without the chat history.
    
    Rules:
    - Only output the rewritten question and nothing else
    - If this is the first question or doesn't need context, return it as-is
    - Ensure the rewritten question is clear and specific
    - Focus on Data Structures and Algorithms context when relevant
      `,
    },
 });
 
 History.pop()
 
 return response.text
}

async function chatting(question)
{
    console.log("🔄 Processing your question...");
    
    const queries=await transformQuery(question);
    
    if (queries.trim() !== question.trim()) {
        console.log(`📝 Interpreted as: "${queries.trim()}"`);
    }
    
    console.log("🔍 Searching knowledge base...");
    
    const embeddings = new GoogleGenerativeAIEmbeddings({
    apiKey: process.env.GEMINI_API_KEY,
    model: 'text-embedding-004',
    });
 
const queryVector = await embeddings.embedQuery(queries);  
  
const pinecone = new Pinecone();
const pineconeIndex = pinecone.Index(process.env.PINECONE_INDEX_NAME);

const searchResults = await pineconeIndex.query({
    topK: 10,
    vector: queryVector,
    includeMetadata: true,
    });
    
    // console.log(" Debug - Search Results:");
    // console.log(`Found ${searchResults.matches.length} matches`);
    // console.log("Top 3 match scores:", searchResults.matches.slice(0, 3).map(m => m.score));
    
const context = searchResults.matches
                   .map(match => match.metadata.text)
                   .join("\n\n---\n\n");
                   
    console.log("📄 Debug - Context length:", context.length);
    console.log("📄 Debug - Context preview:", context.substring(0, 200) + "...");


    History.push({
    role:'user',
    parts:[{text:queries}]
    })   

    console.log("🤖 Generating response...");

    const response = await ai.models.generateContent({
    model: "gemini-2.0-flash",
    contents: History,
    config: {
      systemInstruction: `You are an expert Data Structure and Algorithm teacher and mentor.
    You will be given a context of relevant information and a user question.
    Your task is to answer the user's question based ONLY on the provided context.
    If the answer is not in the context, you must say "I could not find the answer in the provided document."
    Keep your answers clear, concise, and educational.
    
    When discussing algorithms, mention time/space complexity when relevant.
    For coding problems, provide step-by-step explanations.
      
      Context: ${context}
      `,
    },
   });


   History.push({
    role:'model',
    parts:[{text:response.text}]
  })

  console.log("\n" + "=".repeat(60));
  console.log("📚 DSA Expert Response:");
  console.log("=".repeat(60));
  console.log(response.text);
  console.log("=".repeat(60));
 
}

function displayWelcome() {
  console.log("\n" + "=".repeat(60));
  console.log("🎓 Welcome to DSA Expert Chatbot!");
  console.log("=".repeat(60));
  console.log("Ask me anything about Data Structures and Algorithms!");
  console.log("Type 'quit' or 'exit' to end the conversation.");
  console.log("=".repeat(60));
}

async function main(){
   if (History.length === 0) {
       displayWelcome();
   }
   
   const userProblem = readlineSync.question("\n💭 Ask me anything--> ");
   
   if (userProblem.toLowerCase() === 'quit' || userProblem.toLowerCase() === 'exit') {
       console.log("\n👋 Thanks for using DSA Expert! Happy coding!");
       process.exit(0);
   }
   
   await chatting(userProblem);
   main();
}

process.on('SIGINT', () => {
  console.log("\n\n👋 Thanks for using DSA Expert! Happy coding!");
  process.exit(0);
});

main()