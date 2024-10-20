## Execute instruction
- cd current directory.  
- Execute `bash start.sh data/train_ratings.csv [similarity threshold]`  the parameter **similarity threshold** by defaut is 0.05, you can modify it.  *attention:cause i used the cosine similarity, and i advise you to set up similarity threshold 0.01 - 0.08, please do not use too large threshold*
- And wait some times to print out result, to reduce time, i've saved the cached time, Normally, it would read the cache file directly, if you don't want use directley the cache file, you can remove the name 'out.cache' file.⭐️ But i don't advise you to do that, it could spent lots of time to execute, maybe 10min-40min ⭐️  
- Refer to the Project Analysis Technique implementation Documentation for more technical details [Report.pdf] 🫰  
- ✨✨✨ If something executed wrong, please ensure that the versions are consistent, so please execute the code below `pip install -r requirements.txt`  