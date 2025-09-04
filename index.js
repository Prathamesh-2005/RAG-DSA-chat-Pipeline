//Load The PDF
import { PDFLoader } from '@langchain/community/document_loaders/fs/pdf';
import { RecursiveCharacterTextSplitter } from '@langchain/textsplitters';
import { GoogleGenerativeAIEmbeddings } from '@langchain/google-genai';
import * as dotenv from 'dotenv';
import { Pinecone } from '@pinecone-database/pinecone';
import { PineconeStore } from '@langchain/pinecone';

dotenv.config();

async function indexDocument()
{
    const PDF_PATH='./dsa.pdf';
    const pdfLoader=new PDFLoader(PDF_PATH);
    const rawDocs=await pdfLoader.load();
    console.log("PDF Loaded");
    
    //console.log(rawDocs.length);

    //Chunking Here

    
const textSplitter = new RecursiveCharacterTextSplitter({
    chunkSize: 1000,
    chunkOverlap: 200,
  });
const chunkedDocs = await textSplitter.splitDocuments(rawDocs);
   // console.log(chunkedDocs);
    console.log("Chunking done");
    
   //Vector Embedding

   const embeddings = new GoogleGenerativeAIEmbeddings({
    apiKey: process.env.GEMINI_API_KEY,
    model: 'text-embedding-004',
  });

  // Configure the Database 
//Initialize Pinecone Client
const pinecone = new Pinecone();
const pineconeIndex = pinecone.Index(process.env.PINECONE_INDEX_NAME);

//Embed Chunks and Upload to Pinecone
await PineconeStore.fromDocuments(chunkedDocs, embeddings, {
    pineconeIndex,
    maxConcurrency: 5,
});
  
console.log("DATA Stored successfully");


}
indexDocument();