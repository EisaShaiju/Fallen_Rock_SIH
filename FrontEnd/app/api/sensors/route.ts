// app/api/sensors/route.ts

import { Pool } from 'pg'
import { NextResponse } from "next/server"

// Configure your PostgreSQL connection (use environment variables in a real app)
const pool = new Pool({
  user: "your_db_user",
  host: "your_db_host",
  database: "your_db_name",
  password: "your_db_password",
  port: 5432,
})

// This function handles GET requests to /api/sensors
export async function GET(request: Request) {
  try {
    // Connect to the database and execute the query
    const client = await pool.connect()
    // Assuming your table is named 'sensors'
    const result = await client.query("SELECT * FROM sensors;") 
    client.release()

    // Send the data back as a JSON response
    return NextResponse.json(result.rows)
  } catch (error) {
    console.error("Database query failed:", error)
    // Return an error response
    return NextResponse.json(
      { message: "Internal Server Error: Could not fetch sensor data." },
      { status: 500 }
    )
  }
}