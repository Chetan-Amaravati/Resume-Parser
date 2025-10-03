"""
MongoDB handler for resume parser - saves parsed resumes to database
"""
from pymongo import MongoClient
from pymongo.errors import ConnectionFailure
from datetime import datetime
from typing import Dict, List, Optional
import pandas as pd
from bson import ObjectId

class ResumeDB:
    def __init__(self, mongo_uri: str = "mongodb://localhost:27017", db_name: str = "resume_db"):
        """Initialize MongoDB connection"""
        try:
            self.client = MongoClient(mongo_uri, serverSelectionTimeoutMS=5000)
            # Test connection
            self.client.admin.command('ping')
            self.db = self.client[db_name]
            self.resumes_col = self.db['resumes']
            self.scoring_col = self.db['scoring_history']
            print(f"✓ Connected to MongoDB: {db_name}")
        except ConnectionFailure as e:
            print(f"✗ MongoDB connection failed: {e}")
            self.client = None
            self.db = None
            self.resumes_col = None
            self.scoring_col = None
    
    def is_connected(self) -> bool:
        """Check if MongoDB is connected"""
        return self.client is not None and self.db is not None
    
    def save_resume(self, resume_data: Dict) -> Optional[str]:
        """
        Save a single parsed resume to MongoDB
        Returns: ObjectId as string if successful, None otherwise
        """
        if not self.is_connected():
            print("⚠️ MongoDB not connected. Cannot save resume.")
            return None
        
        try:
            # Add timestamp
            resume_data['uploaded_at'] = datetime.utcnow()
            resume_data['last_updated'] = datetime.utcnow()
            
            # Check if resume already exists (by filename)
            existing = self.resumes_col.find_one({"file": resume_data.get("file")})
            
            if existing:
                # Update existing resume
                result = self.resumes_col.update_one(
                    {"_id": existing["_id"]},
                    {"$set": {**resume_data, "last_updated": datetime.utcnow()}}
                )
                print(f"✓ Updated resume: {resume_data.get('file')}")
                return str(existing["_id"])
            else:
                # Insert new resume
                result = self.resumes_col.insert_one(resume_data)
                print(f"✓ Saved new resume: {resume_data.get('file')}")
                return str(result.inserted_id)
                
        except Exception as e:
            print(f"✗ Error saving resume: {e}")
            return None
    
    def save_resumes_batch(self, df: pd.DataFrame) -> List[str]:
        """
        Save multiple resumes from DataFrame to MongoDB
        Returns: List of ObjectId strings
        """
        if not self.is_connected():
            print("⚠️ MongoDB not connected. Cannot save resumes.")
            return []
        
        saved_ids = []
        for _, row in df.iterrows():
            resume_dict = row.to_dict()
            resume_id = self.save_resume(resume_dict)
            if resume_id:
                saved_ids.append(resume_id)
        
        return saved_ids
    
    def save_scoring_result(self, file: str, jd: str, score_data: Dict) -> Optional[str]:
        """Save scoring results for a resume"""
        if not self.is_connected():
            return None
        
        try:
            scoring_record = {
                "file": file,
                "job_description": jd,
                "score": score_data.get("score"),
                "matched": score_data.get("matched", []),
                "missing": score_data.get("missing", []),
                "scored_at": datetime.utcnow()
            }
            result = self.scoring_col.insert_one(scoring_record)
            return str(result.inserted_id)
        except Exception as e:
            print(f"✗ Error saving scoring result: {e}")
            return None
    
    def get_resume_by_filename(self, filename: str) -> Optional[Dict]:
        """Retrieve a resume by filename"""
        if not self.is_connected():
            return None
        
        resume = self.resumes_col.find_one({"file": filename})
        if resume:
            resume['_id'] = str(resume['_id'])
        return resume
    
    def get_all_resumes(self, limit: int = 100) -> List[Dict]:
        """Get all resumes from database"""
        if not self.is_connected():
            return []
        
        resumes = list(self.resumes_col.find().limit(limit))
        for r in resumes:
            r['_id'] = str(r['_id'])
        return resumes
    
    def get_resumes_dataframe(self) -> pd.DataFrame:
        """Get all resumes as a pandas DataFrame"""
        if not self.is_connected():
            return pd.DataFrame()
        
        resumes = self.get_all_resumes()
        if not resumes:
            return pd.DataFrame()
        
        # Remove MongoDB _id from DataFrame
        for r in resumes:
            if '_id' in r:
                del r['_id']
        
        return pd.DataFrame(resumes)
    
    def delete_resume(self, filename: str) -> bool:
        """Delete a resume by filename"""
        if not self.is_connected():
            return False
        
        try:
            result = self.resumes_col.delete_one({"file": filename})
            return result.deleted_count > 0
        except Exception as e:
            print(f"✗ Error deleting resume: {e}")
            return False
    
    def get_scoring_history(self, filename: str) -> List[Dict]:
        """Get scoring history for a specific resume"""
        if not self.is_connected():
            return []
        
        history = list(self.scoring_col.find({"file": filename}).sort("scored_at", -1))
        for h in history:
            h['_id'] = str(h['_id'])
        return history
    
    def close(self):
        """Close MongoDB connection"""
        if self.client:
            self.client.close()
            print("✓ MongoDB connection closed")