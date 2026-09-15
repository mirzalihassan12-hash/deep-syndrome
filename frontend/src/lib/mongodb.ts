import { MongoClient, type Db } from "mongodb";

const uri = process.env.MONGODB_URI;

declare global {
  var _mongoClientPromise: Promise<MongoClient> | undefined;
}

function getClientPromise(): Promise<MongoClient> {
  if (!uri) {
    throw new Error("Missing MONGODB_URI environment variable");
  }

  if (process.env.NODE_ENV === "development") {
    // Reuse across HMR reloads in dev so we don't open a new connection per edit.
    if (!global._mongoClientPromise) {
      global._mongoClientPromise = new MongoClient(uri).connect();
    }
    return global._mongoClientPromise;
  }

  // Production/Vercel: fresh per cold start, reused across warm invocations
  // of the same lambda instance via this module-level variable.
  if (!cachedProdClientPromise) {
    cachedProdClientPromise = new MongoClient(uri).connect();
  }
  return cachedProdClientPromise;
}

let cachedProdClientPromise: Promise<MongoClient> | undefined;

async function getDb(): Promise<Db> {
  const client = await getClientPromise();
  return client.db(process.env.MONGODB_DB || "deep_syndrome");
}

export async function getDoctorsCollection() {
  const db = await getDb();
  return db.collection("doctors");
}

export async function getConfirmedSamplesCollection() {
  const db = await getDb();
  return db.collection("confirmed_samples");
}
