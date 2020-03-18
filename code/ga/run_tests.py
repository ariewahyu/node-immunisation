from . import tests
import sys

def runTest(fname):
    try:
        testMethod = getattr(tests, fname)
        testMethod()
    except Exception as ex:
        print('Test {} failed - {}: {}'.format(fname, type(ex).__name__, ex))
        raise ex
    else:
        print('Test {} passed'.format(fname))


if(len(sys.argv) == 1):
    print("Running all tests")
    for fname in dir(tests)[::-1]:
        if(fname.startswith('test_')):
            runTest(fname)
else:
    print("Running test {}".format(sys.argv[1]))
    runTest(sys.argv[1])
