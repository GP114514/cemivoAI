using System;
using System.Diagnostics;
using System.IO;
using System.Windows.Forms;
static class CemivoLauncher {
    [STAThread] static void Main() {
        string root=AppDomain.CurrentDomain.BaseDirectory;
        string script=Path.Combine(root,"screen.ps1");
        if(!File.Exists(script)){MessageBox.Show("Keep cemivoAI.exe in the application folder.","cemivoAI");return;}
        try {
            var start=new ProcessStartInfo("powershell.exe", "-NoProfile -ExecutionPolicy Bypass -WindowStyle Hidden -File \""+script+"\"");
            start.WorkingDirectory=root;start.UseShellExecute=false;start.CreateNoWindow=true;
            using(var process=Process.Start(start)) {
                process.WaitForExit();
                if(process.ExitCode!=0)MessageBox.Show("Unable to start. Check model setup and work/screen-server-error.log.","cemivoAI");
            }
        } catch(Exception error){MessageBox.Show(error.Message,"cemivoAI");}
    }
}
