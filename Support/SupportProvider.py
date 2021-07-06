import inspect

class SupportProvider(object):

    _class_name:str = None

    def __init__(self) -> None:  
        try:

            print("Init SupportProvider class")
            self._class_name = __class__.__name__

        except Exception as ex:
            self.ExceptMessage(classname = self._class_name,
                               funcname=inspect.currentframe().f_code.co_name,
                               exception=ex)

    def ExceptMessage(classname:str, 
                      funcname:str, 
                      exception:any) -> None:

        if (classname == None or classname is ''):
            return 'ExceptMessage failed by missing class name value!'

        if (funcname == None or funcname is ''):
            return 'ExceptMessage failed by missing function name value!'

        if (exception == None):
            return 'ExceptMessage failed by missing exception object value!'

        template = "An exception of type {exception} occurred in [{cname}.{fname}]. Arguments:\n{rest!r}"
        message = template.format(exception = type(exception).__name__, 
                                  cname = classname, 
                                  fname = funcname, 
                                  rest = exception.args)
        print(message)


if __name__ == "__main__":
    SupportProvider()