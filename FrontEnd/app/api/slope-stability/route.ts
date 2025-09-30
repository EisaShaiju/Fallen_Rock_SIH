// app/api/slope-stability/route.ts

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
    // Query for the most recent slope stability data.
    // This assumes your data is in a table and structured as a single JSON object.
    const result = await client.query("SELECT data FROM slope_stability_predictions ORDER BY last_calculated DESC LIMIT 1;")
    client.release()

    // The data is likely stored in a single row/column, so we return the first result.
    if (result.rows.length > 0) {
      return NextResponse.json(result.rows[0].data)
    } else {
      throw new Error("No slope stability data found.")
    }
  } catch (error) {
    console.error("Database query failed:", error)
    return NextResponse.json(
      { message: "Internal Server Error: Could not fetch slope stability data." },
      { status: 500 }
    )
  }
}