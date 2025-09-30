// app/api/rockfall/route.ts
import { Pool } from "pg"
import { NextResponse } from "next/server"

const pool = new Pool({
  user: "your_db_user",
  host: "your_db_host",
  database: "your_db_name",
  password: "your_db_password",
  port: 5432,
})

export async function GET(request: Request) {
  try {
    const client = await pool.connect()
    // Query your rockfall data table
    const result = await client.query("SELECT data FROM rockfall_predictions ORDER BY last_calculated DESC LIMIT 1;")
    client.release()
    
    if (result.rows.length > 0) {
      return NextResponse.json(result.rows[0].data)
    } else {
      throw new Error("No rockfall data found.")
    }
  } catch (error) {
    // Log the error for debugging on the server
    console.error("Database query failed:", error)
    // Return a JSON response with a 500 status code
    return NextResponse.json(
      { message: "Internal Server Error: Could not fetch rockfall data." },
      { status: 500 }
    )
  }
}