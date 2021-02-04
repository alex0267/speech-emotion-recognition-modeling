## DVC

### What is it ? 

Dvc is a program installed on your desktop. His main function is to help a team to keep track of large datasets. 

### Why use it ? 

Version control system are not well suited to keep track of large files. 
For example, GitHub limits the size of files allowed in repositories, and will block a push to a repository if the files are larger than the maximum file limit.

### How does it work ? 

You can see it as a version control for data, a git for data.
This analogy is so vivid that a lot of git commands can be used as is in DVC. 
For example :
- to add a file to the staging environment in git, you use the following command :

    ```git add myfile.txt```
    
- to add a datafile to the staging environment in git, you use the following command :

    ```dvc add myfile.csv```
    
To make you understand the workflow used in dvc, I have first to describe the function of a special 
file in git projects, the *.gitignore* file. It contains all the files that git should ignore (those files will never go to the staging phase). 

Thus, the overall workflow is the following : 

- add a data file to dvc with the command 

  ```dvc add myfile.csv``` 

- ignore the data file in git and add a reference to the data file named myfile.csv.dvc 
  
  ```dvc add .gitignore myfile.csv.dvc```
  
- push the data in your data specific backend (gdrive, google cloud storage, s3 or whatever you use) 
   
   ```dvc push```

More details on the workflow related to data :

- [data-versioning](https://dvc.org/doc/start/data-versioning)
- [data-access](https://dvc.org/doc/start/data-access)


Here is a general introductory [Video](https://www.youtube.com/watch?v=kLKBcPonMYw)

To register a specific backend storage, you can go to [this](https://dvc.org/doc/command-reference/remote/add) page




