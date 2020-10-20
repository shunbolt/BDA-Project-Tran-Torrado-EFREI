if len(sys.argv) == 1 and int(sys.argv[1]) > 1 and int(sys.argv[1]) < 100 :
    estimators = 10
    print("No argument, set estimators to " + str(estimators))
elif (len(sys.argv) == 1) and int(sys.argv[1]) > 1 and int(sys.argv[1]) < 100 :
    estimators = int(sys.argv[1])
    print("good argument, set estimators to " + str(estimators))
else :
    estimators = 10
    print("bad argument, set estimators to " + str(estimators))