# PostgreSQL Setup Guide

This guide provides step-by-step instructions to set up a PostgreSQL database for your project via the command line interface (CLI).

## Step 1: Install PostgreSQL

### On Ubuntu
```bash
sudo apt update
sudo apt install postgresql postgresql-contrib
```

### On macOS (Using Homebrew)
```bash
brew update
brew install postgresql
brew services start postgresql
```

### On Windows
Download and install PostgreSQL from the [official website](https://www.postgresql.org/download/). During installation, make a note of the password you set for the `postgres` user.

---

## Step 2: Start the PostgreSQL Service

Ensure the PostgreSQL service is running:

### On Ubuntu
```bash
sudo systemctl start postgresql
sudo systemctl enable postgresql
```

### On macOS
```bash
brew services start postgresql
```

---

## Step 3: Access the PostgreSQL CLI

Switch to the PostgreSQL user and access the `psql` shell:
```bash
sudo -i -u postgres
psql
```

If you are on Windows, you can access `psql` from the command line by searching for "SQL Shell (psql)" in the Start menu.

---

## Step 4: Create a Database

In the `psql` shell, create a new database:
```sql
CREATE DATABASE optionsdb;
```

---

## Step 5: Create a New User

Create a new user with a secure password:
```sql
CREATE USER volastudio_user WITH PASSWORD 'secure_password';
```

Replace `secure_password` with a strong password of your choice.

---

## Step 6: Grant Privileges

Grant the necessary privileges to the new user for the database:
```sql
GRANT ALL PRIVILEGES ON DATABASE optionsdb TO volastudio_user;
```

---

## Step 7: Verify the Connection

Exit the `psql` shell by typing:
```bash
\q
```

Test the connection to the database using the new user:
```bash
psql -U volastudio_user -d optionsdb -h localhost
```

If the connection is successful, your PostgreSQL setup is complete.

---

## Step 8: Update Project Configuration

Update your project's `.env` file or configuration settings with the database connection string:
```env
POSTGRES_URI=postgresql://volastudio_user:secure_password@localhost:5432/optionsdb
```

---

## Additional Commands

### List Databases
```sql
\l
```

### List Users
```sql
\du
```

### Connect to a Database
```sql
\c optionsdb
```

### List Tables in the Current Database
```sql
\dt
```

---

## Troubleshooting

### Check PostgreSQL Service Status
#### On Ubuntu:
```bash
sudo systemctl status postgresql
```

#### On macOS:
```bash
brew services list
```

### Common Errors
- **Error: Role does not exist**: Ensure you created the user and granted privileges.
- **Error: Could not connect to server**: Verify the PostgreSQL service is running and check your connection string.

For more assistance, refer to the [official PostgreSQL documentation](https://www.postgresql.org/docs/).

