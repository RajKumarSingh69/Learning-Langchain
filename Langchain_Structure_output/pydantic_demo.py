from pydantic import BaseModel,EmailStr,Field
from typing import Optional
class Student(BaseModel):
    name:str='Raj' # thia ia by-default value and if any value comes then it will replaced by that value
    age: Optional[int]=None # in this , with the help of pydantic Optional ,if any value dosn't provided then it set as None
    email:EmailStr
    cgpa: float=Field(gt=0,lt=10,default=5,description=" A decimal value representing the cgpa of the student")


new_student={'name':'Raj Kumar','age':'65','email':'Rajkr@gmal.com','cgpa':8} # pydantci by default did implicit type conversion
student=Student(**new_student)
print(student)
print(type(student))