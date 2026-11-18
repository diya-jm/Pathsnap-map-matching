"""Launch one side of the demonstration with an ordinary Python installation."""
import argparse
from functools import partial
from http.server import SimpleHTTPRequestHandler,ThreadingHTTPServer
from pathlib import Path
import os

if __name__=="__main__":
    parser=argparse.ArgumentParser()
    parser.add_argument("component",choices=["api","ui"])
    parser.add_argument("--host",default="127.0.0.1")
    parser.add_argument("--port",type=int)
    parser.add_argument("--trips",type=Path,default=Path("data/Student folder/student_test"))
    args=parser.parse_args()
    if args.component=="api":
        import uvicorn
        os.environ["PATHSNAP_TRIPS"]=str(args.trips)
        uvicorn.run("api:app",host=args.host,port=args.port or 8000,access_log=True)
    else:
        handler=partial(SimpleHTTPRequestHandler,directory=str(Path(__file__).parent/"ui"))
        port=args.port or 8501
        print(f"Open http://{args.host}:{port}",flush=True)
        ThreadingHTTPServer((args.host,port),handler).serve_forever()

